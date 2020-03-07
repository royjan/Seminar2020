import logging


class Logger:

    def __init__(self):
        self.writer = self.create_logger()

    @staticmethod
    def create_logger(verbose: int = logging.DEBUG):
        logging.basicConfig(format='%(asctime)s %(message)s')
        log = logging.getLogger()
        log.setLevel(verbose)
        return log

    def set_logger_severity(self, verbose: str):
        """
        :param verbose: what is the output level the client wants to see
        """
        current_severity = logging._levelToName[self.writer.level]
        next_severity_str = verbose.upper()
        next_severity = logging._nameToLevel.get(next_severity_str, -1)
        if next_severity == -1:
            self.writer.error(f"Can't find this level severity: {next_severity_str}")
            return
        if next_severity_str == current_severity:
            return
        else:
            self.writer.error(f"Change level from {current_severity} to {next_severity_str}")
            self.writer.setLevel(next_severity)


logger = Logger()
writer = logger.writer
