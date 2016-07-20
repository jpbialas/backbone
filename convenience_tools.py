import progressbar

def v_print(myStr, verbose):
	if verbose:
		print(myStr)

def custom_progress():
	return progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])

