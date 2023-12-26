import redis

class redis_util():
    def __init__(self) -> None:
        self.pool = redis.ConnectionPool(host='10.112.71.164', port=6379, db=0)
        
    
    def get(self, key):
        redis = redis.Redis(connection_pool=self.pool)
        return redis.get(key)
    
    def set(self, key, value):
        redis = redis.Redis(connection_pool=self.pool)
        redis.set(key, value)
        redis.expire(key, 5)
    
    def delete(self, key):
        redis = redis.Redis(connection_pool=self.pool)
        self.redis.delete(key)
    

class where2comm_comm():
    def __init__(self) -> None:
        self.redis = redis_util()
    
    def put_state(self, id, state):
        key = 'where2comm_' + id
        self.redis.set(key, state)
    
    def get_state(self, id):
        key = 'where2comm_' + id
        return self.redis.get(key)