"""
OpenTranslate - Decentralized Multilingual Translation Network for Scientific Knowledge
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="opentranslate",
    version="0.1.0",
    author="OpenTranslate Team",
    author_email="contact@opentranslate.org",
    description="Revolutionary decentralized multilingual translation network for scientific knowledge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opentranslate/opentranslate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Education",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: Japanese",
        "Natural Language :: Korean",
        "Natural Language :: Russian",
        "Natural Language :: Spanish",
        "Natural Language :: French",
        "Natural Language :: German",
        "Natural Language :: Arabic",
        "Natural Language :: Portuguese",
        "Natural Language :: Vietnamese",
        "Natural Language :: Bengali",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.9",
            "flake8>=4.0",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
            "sphinx-multiversion>=0.2",
        ],
        "blockchain": [
            "web3>=5.0",
            "eth-account>=0.5",
            "eth-typing>=2.0",
            "eth-utils>=1.0",
            "py-solc-x>=1.0",
        ],
        "ai": [
            "torch>=1.9",
            "transformers>=4.0",
            "sentencepiece>=0.1.96",
            "sacrebleu>=2.0",
            "nltk>=3.6",
            "spacy>=3.0",
            "fastai>=2.0",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8",
            "sqlalchemy>=1.4",
            "alembic>=1.7",
            "redis>=4.0",
        ],
        "web": [
            "streamlit>=1.0",
            "plotly>=5.0",
            "dash>=2.0",
            "gradio>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "opentranslate=opentranslate.cli:cli",
            "opentranslate-api=opentranslate.api:main",
            "opentranslate-web=opentranslate.web:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/opentranslate/opentranslate/issues",
        "Source": "https://github.com/opentranslate/opentranslate",
        "Documentation": "https://docs.opentranslate.org",
        "Community": "https://community.opentranslate.org",
        "Blog": "https://blog.opentranslate.org",
        "Twitter": "https://twitter.com/OpenTranslate",
        "Discord": "https://discord.gg/opentranslate",
    },
) 