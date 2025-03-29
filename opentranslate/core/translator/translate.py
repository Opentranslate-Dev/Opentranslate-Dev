#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import requests
import pyperclip
import sys
from typing import Optional

class Translator:
    def __init__(self):
        self.api_url = "http://localhost:8000/api/v1/translate"
        
    def translate(self, text: str, from_lang: str, to_lang: str) -> Optional[str]:
        """
        Translate text using the OpenTranslate API
        
        Args:
            text: Text to translate
            from_lang: Source language code
            to_lang: Target language code
            
        Returns:
            Translated text or None if translation fails
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "text": text,
                    "source_lang": from_lang,
                    "target_lang": to_lang
                }
            )
            response.raise_for_status()
            return response.json()["translation"]
        except Exception as e:
            print(f"Translation failed: {e}", file=sys.stderr)
            return None

def main():
    parser = argparse.ArgumentParser(description="Translate text using OpenTranslate")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--from-lang", default="auto", help="Source language code")
    parser.add_argument("--to-lang", required=True, help="Target language code")
    parser.add_argument("--copy", action="store_true", help="Copy result to clipboard")
    
    args = parser.parse_args()
    
    translator = Translator()
    result = translator.translate(args.text, args.from_lang, args.to_lang)
    
    if result:
        print(result)
        if args.copy:
            pyperclip.copy(result)
            print("Result copied to clipboard", file=sys.stderr)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 