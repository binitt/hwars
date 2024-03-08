OS=$(uname | perl -ne 'if (m/cygwin/i) { print "WINDOWS" } else { print "LINUX"; }')

if [ "$OS" == "WINDOWS" ]; then
  source $PYTHON_ENV/Scripts/activate
else
  source $PYTHON_ENV/bin/activate
fi

pip install --force-reinstall dist/*.whl
