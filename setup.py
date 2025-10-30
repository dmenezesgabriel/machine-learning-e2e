import tomllib  # Python 3.11+ built-in TOML parser

from setuptools import find_packages, setup


def get_project_metadata():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    return {
        "name": project.get("name", "unknown"),
        "version": project.get("version", "0.0.0"),
        "description": project.get("description", ""),
        "readme": project.get("readme", "README.md"),
        "python_requires": project.get("requires-python", ">=3.8"),
        "install_requires": project.get("dependencies", []),
    }


meta = get_project_metadata()

setup(
    name=meta["name"],
    version=meta["version"],
    description=meta["description"],
    long_description=open(meta["readme"]).read(),
    long_description_content_type="text/markdown",
    python_requires=meta["python_requires"],
    packages=find_packages(),
    install_requires=meta["install_requires"],
)
