# Developer Guidance

## Using YAPF and Other Formatting

YAPF is an open-source formatter for Python. It supports styles such as PEP8 or Google Style. For detailed description, please see [YAPF](https://github.com/google/yapf)

YAPF currently supports Python 2.7 and 3.6.4+. To install YAPF, run

``$ sudo pip install yapf``

You can also use Anaconda to install:

``$ conda install -c conda-forge yapf``

In order to format your python script, simply run, 

``$ yapf <python file>``

The default style is PEP8

And if you want to see the differences before and after, use

``$ yapf -d <python file>``

For more usage please see

``yapf -h``

If you are using VS code and want to automatically format the file on save, go to "File-Preferences-Settings", search for "Python Formatting Provider", select `yapf` as your formatting provider. Then search for "Format On Save", toggle the option. 

If you still cannot format on save, it might be because you have previously installed other formatting extensions. Try selecting a segment of the code, right-click it and choose "Format Document", if the prompt asks you about the formatter, choose Python.

YAPF will NOT sort your imports, to do that, press Ctrl+Shift+P, then search for sort imports. The packages will be sorted according to PEP8 standards.

## Git Cooperation

If you want to make contributions to the repository, please clone the repo to your local address, create your own branch, make the changes, commit and push to the remote repository. Please check the correctness of your implementations before pushing.

After pushing your branch, create a new pull request. Other developers will see the request and leave suggestions. You may discuss with them under the request. 

If you want to make further modifications, make them and push again. Others will see the update under your pull request. If everything goes well, changes will be merged into `master` branch and the request will be closed.

## Python Environment Setting

Please check your interpreter under VS code to make sure the environment is set up correctly. To do that, press Ctrl+Shift+P, then search for python interpreter, select the conda environment you installed. 

