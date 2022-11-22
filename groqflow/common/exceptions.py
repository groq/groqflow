import groqflow.common.printing as printing


class GroqFlowError(Exception):
    """
    Indicates the user did something wrong with groqit
    """

    def __init__(self, msg):
        super().__init__()
        printing.log_error(msg)


class GroqitCacheError(GroqFlowError):
    """
    Indicates ambiguous behavior from
    when a build already exists in the GroqFlow cache,
    but the model, inputs, or args have changed thereby invalidating
    the cached copy of the model.
    """


class GroqitEnvError(GroqFlowError):
    """
    Indicates to the user that the required tools are not
    available on their PATH.
    """


class GroqitArgError(GroqFlowError):
    """
    Indicates to the user that they provided invalid arguments to
    groqit()
    """


class GroqitStageError(Exception):
    """
    Let the user know that something went wrong while
    firing off a groqit() Stage.

    Note: not overloading __init__() so that the
    attempt to print to stdout isn't captured into
    the Stage's log file.
    """


class GroqitStateError(Exception):
    """
    Raised when something goes wrong with State
    """


class GroqitIntakeError(Exception):
    """
    Let the user know that something went wrong during the
    initial intake process of analyzing a model.
    """


class GroqModelEnvError(GroqFlowError):
    """
    Indicates to the user that the required tools are not
    available on their PATH when running GroqModel.
    """


class GroqModelRuntimeError(GroqFlowError):
    """
    Let the user know that something went wrong during runtime
    execution on the GroqCard accelerators.
    """


class GroqModelRemoteError(GroqFlowError):
    """
    Let the user know that something went wrong during runtime
    execution on the GroqCard accelerators when using remote backend.
    """


class GroqModelArgError(GroqFlowError):
    """
    Indicates to the user that they provided invalid arguments to
    GroqModel()
    """


class GroqFlowIOError(GroqFlowError):
    """
    Indicates to the user that an input/output operation failed,
    such trying to open a file.
    """
