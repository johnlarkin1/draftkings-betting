echo "Running black..."
black src/
echo "Running pylint..."
pylint src/
echo "Running mypy..."
mypy src/
