def cache_by_entity_id(func):
    cache = dict()

    def wrapper(*args, **kwargs):
        entity = args[-1]
        if entity['id'] in cache:
            return cache[entity['id']]
        else:
            value = func(*args, **kwargs)
            cache[entity['id']] = value
            return value

    return wrapper
