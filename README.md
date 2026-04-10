## Challenges
1. A document is composed of multiple languages, so I need to create a percentage representation.
2. There are many languages, should I identify only the top most popular language
3. Supported documents is pdf.
pdf can be of two types images and text.    


Broadly speaking, there are about 7,168 living languages in the world today. However, linguistic diversity is heavily skewed: just 23 languages account for more than half of the world's population.
I'll take the top 10 languages


The "Middle" Way: Script Detection (Unicode)
Before identifying the language (e.g., "Is this Spanish or Italian?"), you can identify the Script (e.g., "Is this Latin or Cyrillic?"). Every character has a "Script" property in the Unicode Standard.

Unicode Ranges: Characters are organized into blocks. For example:

U+0000 to U+007F: Basic Latin (English, etc.)

U+0600 to U+06FF: Arabic

U+0900 to U+097F: Devanagari (Hindi)

How it works: You iterate through the characters of your string and check which Unicode block they fall into. If 90% of characters are in the Devanagari block, you know the script is Devanagari, which narrows the language down to Hindi, Marathi, Nepali, etc.


Language and scripts are different. 
So simple identification of script will not work if you check the unicode ranges.


Rank,Language,Total Speakers,Primary Script
1,English,1.5 Billion,Latin
2,Mandarin Chinese,1.2 Billion,Han (Simplified/Traditional)
3,Hindi,629 Million,Devanagari
4,Spanish,590 Million,Latin
5,French,416 Million,Latin
6,Modern Arabic,335 Million,Arabic
7,Portuguese,282 Million,Latin
8,Bengali,278 Million,Bengali-Assamese