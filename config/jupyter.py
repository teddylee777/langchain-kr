c = get_config()

c.NotebookApp.allow_root = True
c.NotebookApp.allow_origin = 'http://192.168.0.1.0/24'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 9190

