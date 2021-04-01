class SimplexError(Exception):
    pass


class NoSolution(SimplexError):
    pass


class IncorrectInitialSolution(SimplexError):
    pass


class UnboundFunction(SimplexError):
    pass
