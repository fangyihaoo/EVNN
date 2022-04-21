import logging
import os
import sys
from datetime import datetime
# path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(path)



def logger_init(log_file_name: str = 'monitor',
                log_level: int = logging.DEBUG,
                log_dir: str = '../log/',
                only_file: bool = False
                ):
    """logger helper function
    Args:
        log_file_name (str, optional): name of the log file. Defaults to 'monitor'.
        log_level (int, optional): level of log file information. Defaults to logging.DEBUG.
        log_dir (str, optional): path of log file. Defaults to './log/'.
        only_file (bool, optional): whether export to str. Defaults to False.
    Returns:
        _type_: logging object
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    # formatter = '%(asctime)s - %(module)s - %(name)s - %(levelname)s - %(message)s'
    formatter = '%(module)s - %(name)s - %(levelname)s - %(message)s'
    
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )
        
    return logging.getLogger(__name__)

if __name__ == "__main__":
    logger = logger_init()
    logger.info('Test')