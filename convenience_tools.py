import progressbar
#import matplotlib.pyplot as plt

def v_print(myStr, verbose):
    '''
        myStr: String to print
        verbose: Boolean indicating whether or not to print
        OUTPUT:
            Prints myStr only if verbose is true
    '''
    if verbose:
        print(myStr)

def custom_progress():
    '''
        RETURNS:
            instance of custom progress bar rapper
    '''
    return progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])

def show_img(img, title = ""):
    fig = plt.figure()
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.imshow(img)
    plt.title(title), plt.xticks([]), plt.yticks([])
    return fig