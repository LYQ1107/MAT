class MATError(Exception):
    """Base error for failures that should be visible in a run manifest."""


class ValidationError(MATError, ValueError):
    pass


class MissingAssetError(MATError):
    pass


class RouteNotApprovedError(MATError):
    pass


class IntegrityError(MATError):
    pass


class DependencyUnavailableError(MATError):
    pass


class ProtocolError(MATError):
    pass

