# GroqFlow 🚀

GroqFlow™ is the easiest way to get started with Groq's technology. GroqFlow provides an automated tool flow for compiling machine learning and linear algebra workloads into Groq programs and executing those programs on GroqChip™ processors.

We recommend that your system meets the following hardware requirements:

- To build models: 32GB or more of RAM.
- To run models: 8 GroqChip processors is recommended, especially for larger models.

---

## Installation Guide

Sign-up on [support.groq.com](https://support.groq.com) to download and install GroqWare™ Suite version >=0.9.2.1.

For installation instructions, please have a look at our [Install Guide](docs/install.md).


## Getting Started

To Groq a PyTorch model, simply provide your model and inputs to the `groqit()` function. Once `groqit()` has built your model, you can execute your Groq model the same way you execute a PyTorch model.

<img src="https://github.com/groq/groqflow/raw/main/docs/img/groqflow.gif"  width="800"/>


`groqit()` also works with ONNX files and provides many customization options. You can find more information about how to use groqit() in our [User Guide](docs/user_guide.md).

---

## Navigating GroqFlow

* [demo_helpers](demo_helpers/): Scripts used for GroqFlow demos and proof points.

* [docs](docs/): All information you'd need to be successful with GroqFlow.

* [examples](examples/): Includes various GroqFlow examples.

* [groqflow](groqflow/): The source code for the `groqflow` package.

* [proof_points](proof_points/): Machine learning proof points using GroqFlow.

* [readme.md](readme.md): This readme.

* [setup.py](setup.py): GroqFlow setup script for installation.

## Contributors

GroqFlow development is primarily conducted within Groq's internal repo and is periodically synced to GitHub. This approach means that developer contributions are not immediately obvious in the commit log. The following awesome developers have contributed to GroqFlow (order is alphabetical):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://www.linkedin.com/in/danielholandanoronha"><img src="https://avatars.githubusercontent.com/u/9607530?v=4" width="100px;" alt="Daniel Holanda"/><br /><sub><b>Daniel Holanda</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/jeremyfowers"><img src="https://avatars.githubusercontent.com/u/80718789?v=4" width="100px;" alt="Jeremy Fowers"/><br /><sub><b>Jeremy Fowers</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/levzlotnik"><img src="https://avatars.githubusercontent.com/levzlotnik" width="100px;" alt="Lev Zlotnik"/><br /><sub><b>Lev Zlotnik</b></sub></a><br /></td>
      <td align="center"><a href="https://www.linkedin.com/in/philipcolangelo"><img src="https://lh3.googleusercontent.com/pw/AMWts8CciuaYWKT-YVg86giohRGuQI8Jqm3xYeWlkEh41jO4EuPTSn0FLwHp8m0FfLHLIxJOWOxuBRyppa3blDT_YcKokVFbI6yHBYJ1env5evNRCFUPiIBhIlkOzVKMrMMC7aoTjrBGSk6HWUJ803DvMKFudw=s1426-no?authuser=0" width="100px;" alt="Philip Colangelo"/><br /><sub><b>Philip Colangelo</b></sub></a><br /></td>
      <td align="center"><a href="https://www.linkedin.com/in/ramakrishnansivakumar/"><img src="https://media.licdn.com/dms/image/D5603AQGH0fQ4EWzmnw/profile-displayphoto-shrink_200_200/0/1675440402753?e=1680739200&v=beta&t=RddfJm1WgAgyU3Psj76hKuk6__mfqAxm0BqCGlQPWUg" width="100px;" alt="Ramakrishnan Sivakumar"/><br /><sub><b>Ramakrishnan Sivakumar</b></sub></a><br /></td>
      <td align="center"><a href="https://www.linkedin.com/in/sarah-garrod-massengill-76262728/"><img src="https://media.licdn.com/dms/image/C5603AQGJftJLXlWIBA/profile-displayphoto-shrink_200_200/0/1642705991865?e=1680739200&v=beta&t=sHo-x3xpkuUOTJj_4XWm3KERnqwei_4-QTE2pZwFNp4" width="100px;" alt="Sarah Garrod Massengill"/><br /><sub><b>Sarah Garrod Massengill</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/vgodsoe-groq"><img src="https://avatars.githubusercontent.com/u/105250658?v=4" width="100px;" alt="Victoria Godsoe"/><br /><sub><b>Victoria Godsoe</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification.
Contributions of any kind are welcome!