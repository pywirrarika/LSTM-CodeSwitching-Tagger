import re

def morphVec(word):

    upper, title_case, punctuation, at, digit, alnum = 0.0,0.0,0.0,0.0,0.0,0.0
    apost, endsvowel, accented = 0.0,0.0,0.0
    if word.isupper():
        upper = 1.0
    elif word[0].isupper():
        title_case = 1.0
    if word[0] in "¿¡.,?!":
        punctuation = 1.0
    if "@" in word:
        at = 1.0
    if word.isdigit():
        digit = 1.0
    if word.isalnum():
        alnum = 1.0
    if '"' in word or "'" in word:
        apost = 1.0
    if word[-1].lower() in "aeiouáéíóúàèìòù":
        endsvowel = 1.0

    vowels = len(re.findall(r'[aeiouáéíóúàèìòù]',word, re.IGNORECASE))
    alphas = len(re.findall(r'[a-z]',word, re.IGNORECASE))-vowels
    if alphas != 0.0:
        vowels_conson = vowels / alphas
    else:
        vowels_conson= 0.0

    if len(re.findall(r'[áéíóúàèìòù]',word, re.IGNORECASE)) > 0:
        accented = 1.0

    # Falta suffixes y prefixes...

    return [upper, title_case, punctuation, at, digit, alnum, apost, endsvowel, vowels_conson, accented]

def ngramVec(word):
    return []

def camel(word):
    nword = re.sub(r'[a-z]','x',word)
    nword = re.sub(r'[A-Z]','X',nword)
    return nword



def getFeatureVec(word):
    return morphVec(word) + ngramVec(word)

if __name__ == '__main__':
    print(morphVec("Hola"))
    print(morphVec("@HOLA"))
    print(morphVec("¡HOLA"))
    print(morphVec('¡HOLA"'))
    print(morphVec('alberca'))
    print(morphVec('qué'))
    print(morphVec("123"))

