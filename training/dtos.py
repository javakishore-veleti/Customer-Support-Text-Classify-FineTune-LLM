class TrainingReqDTO:

    """
       Loads every worksheet in each Excel file into DataFrames.

       Returns:
           List of dicts like:
           [
               {
                   "file_name": "aws_service_training_dataset.xlsx",
                   "sheets": {
                       "Amazon_S3": <DataFrame>,
                       "Amazon_EC2": <DataFrame>,
                       ...
                   }
               },
               ...
           ]
       """
    def __init__(self):
        self.ctx_data = {}
        self.training_data_excel_filePath = ""
        self.training_data_excel_file_names = []
        self.training_data_dataframes = []  # List of dicts: {file_name: {sheet_name: DataFrame}}


class TrainingResDTO:
    def __init__(self):
        self.ctx_data = {}
