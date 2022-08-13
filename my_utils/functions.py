#This is all functions

def draw_loss( x, y, path, title=None, xlabel='EPOCH', ylabel='LOSS', type_line='.-'):
    import matplotlib.pyplot as plt
    
    plt.plot(x, y, type_line)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.cla()