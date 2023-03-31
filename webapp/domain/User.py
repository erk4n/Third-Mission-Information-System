class User:
    
    @staticmethod
    def empty_user ():
        return User("")

    def __init__(self, user: str):
        self.user = user

    @classmethod
    def from_dict(cls, doc: dict) -> 'User':
        return cls(doc['user'])
