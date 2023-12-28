import redis

class RedisUtils():
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