# Import specific modules explicitly rather than eagerly loading all submodules.
# data_fetch imports astroquery which makes network calls on import — keep it lazy.
