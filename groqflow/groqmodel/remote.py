import base64
import hashlib
import io
import os
from typing import Any, Collection, Dict, List, Tuple
from dataclasses import dataclass
import requests
import numpy as np
import groqflow.common.build as build


@dataclass
class UploadUrls:
    iops: List[str]
    inputs: List[Dict[str, str]]
    use_cached_iop: bool


@dataclass
class DownloadUrls:
    outputs: List[Dict[str, str]]


# NOTE: frozen=True because mutations between upload and run wouldn't be caught
# otherwise (objects are passed by reference)
@dataclass(frozen=True)
class RemoteGroqModel:
    user_name: str
    build_name: str
    iop_path: str
    num_chips: int
    num_iterations: int
    input_batch: List[Dict[str, np.ndarray]]
    input_names: List[str]
    post_check_remote_cache_endpoint: str
    post_upload_urls_endpoint: str
    post_execute_endpoint: str

    def _serialize(self, data: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        buffer.seek(0)
        return buffer.read()

    def _deserialize(self, data_raw: bytes) -> np.ndarray:
        buffer = io.BytesIO(data_raw)
        buffer.seek(0)
        return np.load(buffer, allow_pickle=False)

    def _upload_helper(self, url: str, data_raw: bytes) -> None:
        # TODO:(epatrick): error handling
        _response = requests.put(
            url,
            headers={"Content-Type": "application/octet-stream"},
            data=data_raw,
        )

    def _download_helper(self, url: str) -> bytes:
        response = requests.get(
            url,
            headers={"Content-Type": "application/octet-stream"},
        )
        return response.content

    def _calc_file_md5(self, file: str) -> str:
        with open(file, "rb") as iop:
            data_bytes = iop.read()
        md5_bytes = hashlib.md5(data_bytes).digest()
        return base64.b64encode(md5_bytes).decode("utf-8")

    def _calc_iop_md5s(self) -> List[str]:
        iop_files = (
            [f"{self.iop_path}/output.iop"]
            if self.num_chips == 1
            else [f"{self.iop_path}/output.{i}.iop" for i in range(self.num_chips)]
        )
        return list(map(self._calc_file_md5, iop_files))

    def check_remote_cache(self) -> bool:
        iop_md5s = self._calc_iop_md5s()
        response = requests.post(
            self.post_check_remote_cache_endpoint,
            json={
                "user_name": self.user_name,
                "build_name": self.build_name,
                "num_chips": self.num_chips,
                "iop_md5s": iop_md5s,
            },
        )

        if not response.ok:
            # NOTE: we may choose to ignore the exception and pretend the cache
            #       returned false but that should be done at the callsite
            raise Exception(
                f"error status code: {response.status_code}, message: {response.text}"
            )

        body = response.json()
        cache_hit: bool = body["cache_hit"]
        return cache_hit

    def get_upload_urls(self, skip_iop_urls: bool = False) -> UploadUrls:
        response = requests.post(
            self.post_upload_urls_endpoint,
            json={
                "user_name": self.user_name,
                "build_name": self.build_name,
                "num_chips": self.num_chips,
                "input_names": self.input_names,
                "batch_size": len(self.input_batch),
                "skip_iop_urls": skip_iop_urls,
            },
        )

        if not response.ok:
            raise Exception(
                f"error status code: {response.status_code}, message: {response.text}"
            )

        body = response.json()

        input_urls: List[Dict[str, str]] = body["input_urls"]
        iop_urls: List[str] = [] if skip_iop_urls else body["iop_urls"]

        return UploadUrls(
            iops=iop_urls, inputs=input_urls, use_cached_iop=skip_iop_urls
        )

    def _upload_batch(
        self, input_batch: Dict[str, np.ndarray], upload_urls_batch: Dict[str, str]
    ) -> None:
        for input_name, input_data in input_batch.items():
            input_url = upload_urls_batch[input_name]
            input_raw = self._serialize(input_data)
            self._upload_helper(input_url, input_raw)

    def upload(self, upload_urls: UploadUrls) -> None:
        # TODO: error handling
        if not upload_urls.use_cached_iop:
            if self.num_chips == 1:
                iop_files = [f"{self.iop_path}/output.iop"]
            else:
                iop_files = [
                    f"{self.iop_path}/output.{i}.iop" for i in range(self.num_chips)
                ]

            for iop_file, iop_url in zip(iop_files, upload_urls.iops):
                with open(iop_file, "rb") as iop:
                    self._upload_helper(iop_url, iop)

        for batch_index, input_batch in enumerate(self.input_batch):
            self._upload_batch(input_batch, upload_urls.inputs[batch_index])

    def _execute(self) -> Tuple[DownloadUrls, Dict[str, Any]]:
        response = requests.post(
            self.post_execute_endpoint,
            json={
                "user_name": self.user_name,
                "build_name": self.build_name,
                "num_chips": self.num_chips,
                "input_names": self.input_names,
                "batch_size": len(self.input_batch),
                "num_iterations": self.num_iterations,
            },
        )

        if not response.ok:
            raise Exception(
                f"error status code: {response.status_code}, message: {response.text}"
            )

        body = response.json()
        output_urls = body["output_urls"]
        stats = body["stats"]

        return DownloadUrls(outputs=output_urls), stats

    def _download(self, download_urls: DownloadUrls) -> List[Dict[str, np.ndarray]]:
        outputs = []
        for output_urls in download_urls.outputs:
            output = {}
            for output_name, output_url in output_urls.items():
                output_raw = self._download_helper(output_url)
                output[output_name] = self._deserialize(output_raw)
            outputs.append(output)
        return outputs

    def run(self) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Invokes this remote groq model.

        Returns: (output_batch, stats)
            An output_batch where output_batch[i] corresponds to input_batch[i]
            A dictionary of stats for how the model ran on TSPs
        """
        download_urls, stats = self._execute()
        output_batch = self._download(download_urls)
        return output_batch, stats


class RemoteClient:
    """
    A client for running TSP models using remote backend
    """

    # Backend URL is the IP of where the remote server is hosted
    # TODO: Replace backend_url by a hostname
    def __init__(self, backend_url: str = "http://34.125.159.215"):
        self.post_check_remote_cache_endpoint = f"{backend_url}/storage/cache/check"
        self.post_upload_urls_endpoint = f"{backend_url}/storage/upload-urls"
        self.post_execute_endpoint = f"{backend_url}/execute"
        self.user_name = os.getlogin()

    def upload(
        self,
        user_name: str,
        build_name: str,
        compile_dir: str,
        num_chips: int,
        input_batch: Collection[Dict[str, np.ndarray]],
        num_iterations: int = 1,
    ) -> RemoteGroqModel:
        """
        A lower level interface to upload a remote groq model ahead of time. You may
        invoke the remote groq model with the returned RemoteGroqModel object.

        You should also use this interface if you want to combine the functionality of
        benchmark and run_abunch.

        Args:
            user_name: Username of the caller
            build_name: Name of the build
            compile_dir: Full path to the directory containing the IOP file(s)
            num_chips: Number of chips for the remote groq model
            input_batch: Data used as input for the remote groq model. Execution
            will be done once per batch
            num_iterations: How many executions the statistics should be averaged over
            (default = 1)

        Returns:
            A RemoteGroqModel object that can be used to invoke the uploaded model
        """

        input_names = [] if len(input_batch) == 0 else list(list(input_batch)[0].keys())
        remote_gm = RemoteGroqModel(
            user_name,
            build_name,
            compile_dir,
            num_chips,
            num_iterations,
            input_batch,
            input_names,
            self.post_check_remote_cache_endpoint,
            self.post_upload_urls_endpoint,
            self.post_execute_endpoint,
        )
        cache_hit = remote_gm.check_remote_cache()
        upload_urls = remote_gm.get_upload_urls(skip_iop_urls=cache_hit)
        remote_gm.upload(upload_urls)
        return remote_gm

    def execute(
        self,
        state: build.State,
        repetitions: int,
    ) -> Dict[str, np.ndarray]:
        """
        Executes a build on the given inputs and returns the outputs.

        Args:
            state: State of the build being executed
            repetitions: Number of times to execute a build
        Returns:
            The outputs of the execution
        """
        inputs_file = state.execution_inputs_file
        inputs_data = np.load(inputs_file, allow_pickle=True)
        latency_file = state.latency_file
        outputs_file = state.outputs_file
        remote_gm = self.upload(
            self.user_name,
            state.config.build_name,
            state.compile_dir,
            state.num_chips_used,
            inputs_data,
            repetitions,
        )
        output_batch, stats = remote_gm.run()
        latency_avg = stats["exec_time_seconds"]["mean"]
        np.save(latency_file, latency_avg)
        outputs_data = output_batch
        np.save(outputs_file, outputs_data, allow_pickle=True)
