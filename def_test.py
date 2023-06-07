
def test():
    code = """
def function(): print(5)
    """
    exec(code, globals())


test()
function()