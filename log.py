import logging
import inspect
from logging import Formatter

class Log(object):

    ##
    # Logging instance
    #
    __logger = None

    ##
    # Error level
    #
    ERROR = logging.ERROR

    ##
    # Warning level
    #
    WARNING = logging.WARNING

    ##
    # Info level
    #
    INFO = logging.INFO

    ##
    # Debug level
    #
    DEBUG = logging.DEBUG

    ##
    # Records a message in a file and/or displays it in the screen.
    #
    # @param cls     Required argument to the class method.
    # @param level   Integer containing the level of the log message. The possible values are defined in the Log class.
    # @param message String containing the message to be recorded.
    #

    @classmethod
    def log(cls, level, message):
        if not cls.__logger:
            cls.__instantiate()

        if cls.__logger:
            try:
                message = ('%s.%s - ' + message) % Log.__get_callers(inspect.stack())
                cls.__logger.log(level, message)
            except Exception as e:
                cls.log(cls.ERROR, 'Unable to record the log. Error: %s.' % e)

    ##
    # Gets the data about the caller of the log method.
    #
    # @param stack Array containing the system calling stack.
    #
    # @return Array containing the caller class name and the caller method, respectively.
    #
    @staticmethod
    def __get_callers(stack):
        caller_class = None
        caller_method = None
        if stack and len(stack) > 1:
            if stack[1][3] == '<module>':
                caller_method = stack[1][0].f_locals.get('__name__')
                caller_class = ((str(stack[1][0].f_locals.get('__file__'))).split('/')[-1]).split('.')[0]
            else:
                caller_method = stack[1][3]
                if 'self' in stack[1][0].f_locals:
                    caller_class = stack[1][0].f_locals.get('self').__class__.__name__
                elif 'cls' in stack[1][0].f_locals:
                    caller_class = stack[1][0].f_locals.get('cls').__name__
                else:
                    caller_class = 'NoneType'
        return caller_class, caller_method

    ##
    # Creates a new instance of Log class.
    #
    # @param cls Required argument to the class method.
    #
    @classmethod
    def __instantiate(cls):
        try:
            log_file_path = '../results/results.log'
            file_level = 'INFO'
            screen_level = 'INFO'
            enable_screen = True

            cls.__logger = logging.getLogger()
            cls.__logger.setLevel(logging.DEBUG)

            if screen_level and (screen_level in logging._levelNames):
                screen_level_int = logging._levelNames[screen_level]
            else:
                raise Exception('Invalid screen level')

            if file_level and (file_level in logging._levelNames):
                file_level_int = logging._levelNames[file_level]
            else:
                raise Exception('Invalid file level')

            log_format = '[%(levelname)-7s - %(asctime)s] %(message)s'
            formatter = Formatter(log_format)

            screen_handler = logging.StreamHandler()
            screen_handler.setLevel(screen_level_int)
            screen_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(file_level_int)
            file_handler.setFormatter(formatter)

            # add the handlers to the logger
            cls.__logger.addHandler(file_handler)
            # see screen level if enabled
            if enable_screen:
                cls.__logger.addHandler(screen_handler)

            cls.log(Log.DEBUG, '##### Log configurations #####')
            cls.log(Log.DEBUG, 'Log file path: %s' % log_file_path)
            if enable_screen:
                cls.log(Log.DEBUG, 'Screen level: %s' % screen_level)
            cls.log(Log.DEBUG, 'File level: %s' % file_level)
            cls.log(Log.DEBUG, '##############################')
        except Exception as e:
            print ('Unable to get/set log configurations. Error: {}.'.format(e.message))
            cls.__logger = None

    @classmethod
    def info(cls, message):
        cls.log(Log.INFO, message)

    @classmethod
    def error(cls, message):
        cls.log(Log.ERROR, message)

    @classmethod
    def warn(cls, message):
        cls.log(Log.WARNING, message)

    @classmethod
    def debug(cls, message):
        cls.log(Log.DEBUG, message)

    ##
    # Prints a system calling stack.
    #
    # @param stack System calling stack to be printed.
    # It must be created by the caller method using the command "inspect.stack()".
    #
    @staticmethod
    def __print_stack(stack):
        print ('############################################')
        i = 0
        while i < len(stack):
            print ('File:', stack[i][1])
            print ('Caller line:', stack[i][2])
            print ('Caller method:', stack[i][3])
            print ('Caller code:', stack[i][4])
            print ('Undefined:', stack[i][5])
            frame = stack[i][0]
            print ('Frame:')
            for k in frame.f_locals.keys():
                print ('\t%s: %s' % (k, frame.f_locals[k]))
            i += 1
        print ('############################################')
