"""Jobs that run a checkpoint on a GPU node. They need torch and transformers,
which the analysis environment does not install, and import nothing else of
this package, so a copy of the file runs on its own."""
