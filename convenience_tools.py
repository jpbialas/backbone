import progressbar

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

