import sys

from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,ex_tb= error_detail.exc_info() 
    file_name = ex_tb.tb_frame.f_code.co_filename

    #It will give all the information regarding the error like file and line number of exception
    error_message="Error occured in python script name[{0}] and line number[{1}] error message[{2}]".format(
        file_name, ex_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
