import random

HANGMANPICS = ['''
    +---+
    |   |
        |
        |
        |
        |
===========''','''
    +---+
    |   |
    O   |
        |
        |
        |
===========''','''
    +---+
    |   |
    O   |
    |   |
        |
        |
===========''','''
    +---+
    |   |
    O   |
   /|   |
        |
        |
===========''','''
    +---+
    |   |
    O   |
   /|\  |
        |
        |
===========''','''
    +---+
    |   |
    O   |
   /|\  |
   /    |
        |
===========''','''
    +---+
    |   |
    O   |
   /|\  |
   / \  |
        |
===========''','''
    +---+
    |   |
   [O   |
   /|\  |
   / \  |
        |
===========''','''
    +---+
    |   |
   [O]  |
   /|\  |
   / \  |
        |
===========''']

words = {'Colors':'red orange yellow green blue indigo violet white black brown'.split(), 'Shapes':'square triangle rectangle circle ellipse rhombus trapezoid chevron pentagon hexagon septagon octagon'.split(), 'Fruits':'apple orange lemon lime pear watermelon grape grapefruit cherry banana cantaloupe mango strawberry tomato'.split(), 'Animals':'bat bear beaver cat cougar crab deer dog donkey duck eagle fish frog goat leech lion lizard monkey moose mouse otter owl panda python rabbit rat shark sheep skunk squid tiger turkey turtle weasel whale wolf wombat zebra'.split()}

def getRandomWord(wordDict):
    wordKey = random.choice(list(wordDict.keys()))
    wordIndex = random.randint(0, len(wordDict[wordKey])-1)
    return [wordDict[wordKey][wordIndex], wordKey]

def displayBoard(HANGMANPICS, missedLetters, correctLetters, secretWord):
    print(HANGMANPICS[len(missedLetters)])
    print()

    print('Missed letters:', end = ' ')
    for letter in missedLetters:
        print(letter, end = ' ')
    print()

    blanks = '_' * len(secretWord)

    for i in range(len(secretWord)):
        if secretWord[i] in correctLetters:
            blanks = blanks[:i] + secretWord[i] + blanks[i+1:]
    
    for letter in blanks:
        print(letter, end = ' ')
    print()

def getGuess(alreadyGuessed):
    while True:
        print('Guess a letter.')
        guess = input().lower()

        if(len(guess)!=1):
            print('Please enter a single letter.')
        elif guess in alreadyGuessed:
            print('You have already guessed that letter. Choose again.')
        elif guess not in 'abcdefghijklmnopqrstuvwxyz':
            print('Please enter a LETTER.')
        else:
            return guess


def playAgain():
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')

print('H A N G M A N')
missedLetters = ''
correctLetters = ''
secretWord, secretKey = getRandomWord(words)
gameIsDone = False

while True:
    print('The secret word is in the set: ' + secretKey)
    displayBoard(HANGMANPICS, missedLetters, correctLetters, secretWord)

    guess = getGuess(missedLetters+correctLetters)

    if guess in secretWord:
        correctLetters = correctLetters + guess

        foundAllLetters = True
        for i in range(len(secretWord)):
            if secretWord[i] not in correctLetters:
                foundAllLetters = False
                break
        if foundAllLetters:
            print('Yes! The secret word is "' + secretWord + '"! You have won!')
            gameIsDone = True
    else:
        missedLetters = missedLetters + guess

        if(len(missedLetters)==len(HANGMANPICS)-1):
            displayBoard(HANGMANPICS, missedLetters, correctLetters, secretWord)

            print('You have run out of guesses!\nAfter ' + str(len(missedLetters)) + ' missed guesses and '+ str(len(correctLetters)) + 'correct guesses, the word was "' + secretWord + '"')
            gameIsDone = True

    if gameIsDone:
        if playAgain():
            missedLetters = ''
            correctLetters = ''
            gameIsDone = False
            secretWord = getRandomWord(words)
        else: 
            break
