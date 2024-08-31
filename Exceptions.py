
class HkcException(Exception):
    def __init__(self,message = ""):
        self.message = message
    
    
class ImageNotLoadedException(HkcException):
    def __init__(self, filename, message="Image could not be loaded"):
        self.filename = filename
        self.message = f"{message}: {filename}"
        super().__init__(self.message)

