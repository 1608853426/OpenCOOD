import redis

class RedisUtils():
    def __init__(self) -> None:
        self.pool = redis.ConnectionPool(host='10.112.71.164', port=6379, db=0, password='soonmachine')
        
    
    def get(self, key):
        redis_client = redis.Redis(connection_pool=self.pool)
        return redis_client.get(key)
    
    def set(self, key, value):
        redis_client = redis.Redis(connection_pool=self.pool)
        redis_client.set(key, value)
    
    def delete(self, key):
        redis_client = redis.Redis(connection_pool=self.pool)
        redis_client.delete(key)
        
if __name__=='__main__':
    redis_utils = RedisUtils()
    redis_utils.set("test", "test_value")
    print(redis_utils.get("test"))