def import_module_from_GitHub(url,
                    saving_path = './Downloads/'):
    """ 
    A function that imports a Python module published on the Internet.

    Parameters
    ----------
    url : str
        Link to a module. Can be GitHub link to module
        or raw.githubusercontent.com or other, which 
        return HTTP get correctly.
    
    saving_path : str
        Path to saving model. 

    Returns
    -------
    module : module
        Then you can use this module
        Result as: import url 
        
    Examples
    --------
    >>> import_from_GitHub(https://github.com/Gabriel-p/minenergy/blob/master/minenergy.py)
       

    Notes
    -------
    1 : Синтаксис
        Хотелось бы from url import func1 чтобы работало. 
    
    """
    
    
    host = url.split('/')[2]
    if host == 'raw.githubusercontent.com':
        pass
    elif host == 'github.com':
        url = url.replace('github.com','raw.githubusercontent.com').replace('blob/','')
    
    local_filename = url.split('/')[-1]
    assert local_filename.split('.')[-1] == 'py'
    
    # === saving file  == = 
    # NOTE the stream=True parameter below
    import requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(saving_path + local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    
    # === import file  == =
    import importlib.util
    spec = importlib.util.spec_from_file_location(local_filename, saving_path + local_filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
    
    
    
# repo 

# import git

# repo = git.Repo.clone_from(
#     'https://github.com/gitpython-developers/GitPython',
#     './git-python')   
    
    
    
