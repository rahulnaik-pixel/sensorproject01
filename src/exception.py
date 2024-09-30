import sys

def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    filename=exc_tb.tb_frame.f_back.f_code.co_filename
    
    error_message='Error occured python script name [{0}] filename [{1}] error message [{2}]'.format(filename,exc_tb.tb_lineno,str(error))
    return error_message






class Custom_exception(Exception):
    def __init__(self,error_message,error_details: sys):
        super().__init__(error_message)
        self.error_message=error_message_details(
            error_message,error_details=error_details
        )
        

def __str__(self):
    return self.error_message
    
    