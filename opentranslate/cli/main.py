"""
OpenTranslate Main Command Line Interface
"""

import click
from typing import Optional
import uvicorn
from pathlib import Path

from ..config.default import settings
from ..api import APIServer
from ..web import WebServer

@click.group()
def cli():
    """OpenTranslate Command Line Tool"""
    pass

@cli.command()
@click.option("--host", default=settings.API_HOST, help="API server host")
@click.option("--port", default=settings.API_PORT, help="API server port")
@click.option("--reload", is_flag=True, help="Enable hot reload")
def serve_api(host: str, port: int, reload: bool):
    """Start API server"""
    uvicorn.run(
        "opentranslate.api:app",
        host=host,
        port=port,
        reload=reload
    )

@cli.command()
@click.option("--host", default=settings.API_HOST, help="Web server host")
@click.option("--port", default=settings.API_PORT + 1, help="Web server port")
@click.option("--reload", is_flag=True, help="Enable hot reload")
def serve_web(host: str, port: int, reload: bool):
    """Start Web server"""
    uvicorn.run(
        "opentranslate.web:app",
        host=host,
        port=port,
        reload=reload
    )

@cli.command()
@click.argument("text")
@click.option("--source-lang", required=True, help="Source language code")
@click.option("--target-lang", required=True, help="Target language code")
@click.option("--domain", help="Text domain")
@click.option("--priority", type=click.Choice(["low", "normal", "high", "urgent"]), default="normal", help="Translation priority")
def translate(text: str, source_lang: str, target_lang: str, domain: Optional[str], priority: str):
    """Translate text"""
    from ..core.translator import TranslationEngine
    
    engine = TranslationEngine()
    result = engine.translate(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        priority=priority
    )
    
    click.echo(f"Translation result: {result['translations'][0]}")

@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--source-lang", required=True, help="Source language code")
@click.option("--target-lang", required=True, help="Target language code")
@click.option("--domain", help="Text domain")
@click.option("--output", type=click.Path(), help="Output file path")
def translate_file(file_path: str, source_lang: str, target_lang: str, domain: Optional[str], output: Optional[str]):
    """Translate file"""
    from ..core.translator import TranslationEngine
    
    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Translate
    engine = TranslationEngine()
    result = engine.translate(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain
    )
    
    # Output results
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result["translations"][0])
        click.echo(f"Translation results saved to: {output}")
    else:
        click.echo(f"Translation result: {result['translations'][0]}")

@cli.command()
@click.argument("text")
@click.option("--top-k", default=1, help="Return top k most likely domains")
def classify(text: str, top_k: int):
    """Classify text domain"""
    from ..ai.models import DomainClassifier
    
    classifier = DomainClassifier()
    result = classifier.classify(text, top_k=top_k)
    
    click.echo(f"Main domain: {result['main_domain']}")
    click.echo(f"Confidence: {result['confidence']:.2f}")
    click.echo("\nDetailed results:")
    for domain, score in result["domain_scores"].items():
        click.echo(f"- {domain}: {score:.2f}")

@cli.command()
@click.argument("source")
@click.argument("target")
@click.option("--domain", help="Text domain")
def validate(source: str, target: str, domain: Optional[str]):
    """Validate translation"""
    from ..ai.models import ValidationModel
    
    validator = ValidationModel()
    result = validator.validate(source, target, domain)
    
    click.echo(f"Status: {result['status']}")
    click.echo(f"Score: {result['score']:.2f}")
    click.echo(f"Threshold: {result['threshold']:.2f}")
    
    if result["details"]:
        click.echo("\nDetailed results:")
        for detail in result["details"]:
            click.echo(f"- Score: {detail['score']:.2f}")
            click.echo(f"   Passed: {'Yes' if detail['passed'] else 'No'}")

@cli.command()
def init():
    """Initialize project"""
    # Create necessary directories
    dirs = [
        "uploads",
        "models",
        "data",
        "logs"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        click.echo(f"Created directory: {dir_name}")
        
    # Create .env file
    env_path = Path(".env")
    if not env_path.exists():
        env_path.touch()
        click.echo("Created .env file")
        
    click.echo("\nInitialization completed!")
    click.echo("Please edit the .env file to configure necessary parameters.")

if __name__ == "__main__":
    cli() 