import { JSONParseException } from './exception'

export function parseJson (str: string) {
  let i = 0

  const value = parseValue()
  expectEndOfInput()
  return value

  function parseObject () {
    if (str[i] === '{') {
      i++
      skipWhitespace()

      const result: any = {}

      let initial = true
      // if it is not '}',
      // we take the path of string -> whitespace -> ':' -> value -> ...
      while (i < str.length && str[i] !== '}') {
        if (!initial) {
          eatComma()
          skipWhitespace()
        }
        const key = parseString()
        if (key === undefined) {
          expectObjectKey()
        } else {
          skipWhitespace()
          eatColon()
          const value = parseValue()
          result[key] = value
          initial = false
        }
      }
      expectNotEndOfInput()
      // move to the next character of '}'
      i++

      return result
    }
  }

  function parseArray () {
    if (str[i] === '[') {
      i++
      skipWhitespace()

      const result = []
      let initial = true
      while (i < str.length && str[i] !== ']') {
        if (!initial) {
          eatComma()
        }
        const value = parseValue()
        result.push(value)
        initial = false
      }
      expectNotEndOfInput()
      // move to the next character of ']'
      i++
      return result
    }
  }

  function parseValue () {
    skipWhitespace()
    const value: any =
      parseString() ??
      parseNumber() ??
      parseObject() ??
      parseArray() ??
      parseKeyword('true', true) ??
      parseKeyword('false', false) ??
      parseKeyword('null', null)
    skipWhitespace()
    return value
  }

  function parseKeyword (name: string, value: boolean | null) {
    if (str.slice(i, i + name.length) === name) {
      i += name.length
      return value
    }
  }

  function skipWhitespace () {
    while (
      str[i] === ' ' ||
      str[i] === '\n' ||
      str[i] === '\t' ||
      str[i] === '\r'
    ) {
      i++
    }
  }

  function parseString () {
    if (str[i] === '"') {
      i++
      let result = ''
      while (i < str.length && str[i] !== '"') {
        if (str[i] === '\\') {
          const char = str[i + 1]
          if (
            char === '"' ||
            char === '\\' ||
            char === '/' ||
            char === 'b' ||
            char === 'f' ||
            char === 'n' ||
            char === 'r' ||
            char === 't'
          ) {
            result += char
            i++
          } else if (char === 'u') {
            if (
              isHexadecimal(str[i + 2]) &&
              isHexadecimal(str[i + 3]) &&
              isHexadecimal(str[i + 4]) &&
              isHexadecimal(str[i + 5])
            ) {
              result += String.fromCharCode(
                parseInt(str.slice(i + 2, i + 6), 16)
              )
              i += 5
            } else {
              i += 2
              expectEscapeUnicode()
            }
          } else {
            expectEscapeCharacter()
          }
        } else {
          result += str[i]
        }
        i++
      }
      expectNotEndOfInput()
      i++
      return result
    }
  }

  function isHexadecimal (char: string) {
    return (
      (char >= '0' && char <= '9') ||
      (char.toLowerCase() >= 'a' && char.toLowerCase() <= 'f')
    )
  }

  function parseNumber () {
    const start = i
    if (str[i] === '-') {
      i++
      expectDigit()
    }
    if (str[i] === '0') {
      i++
    } else if (str[i] >= '1' && str[i] <= '9') {
      i++
      while (str[i] >= '0' && str[i] <= '9') {
        i++
      }
    }

    if (str[i] === '.') {
      i++
      expectDigit()
      while (str[i] >= '0' && str[i] <= '9') {
        i++
      }
    }
    if (str[i] === 'e' || str[i] === 'E') {
      i++
      if (str[i] === '-' || str[i] === '+') {
        i++
      }
      expectDigit()
      while (str[i] >= '0' && str[i] <= '9') {
        i++
      }
    }
    if (i > start) {
      return Number(str.slice(start, i))
    }
  }

  function eatComma () {
    expectCharacter(',')
    i++
  }

  function eatColon () {
    expectCharacter(':')
    i++
  }

  // error handling
  function expectNotEndOfInput () {
    if (i === str.length) {
      throw new JSONParseException('JSON_ERROR_0001 Unexpected End of Input')
    }
  }

  function expectEndOfInput () {
    if (i < str.length) {
      throw new JSONParseException('JSON_ERROR_0002 Expected End of Input')
    }
  }

  function expectObjectKey () {
    throw new JSONParseException('JSON_ERROR_0003 Expecting JSON Key')
  }

  function expectCharacter (expected: string) {
    if (str[i] !== expected) {
      throw new JSONParseException('JSON_ERROR_0004 Unexpected token')
    }
  }

  function expectDigit () {
    if (!(str[i] >= '0' && str[i] <= '9')) {
      throw new JSONParseException('JSON_ERROR_0006 Expecting a digit')
    }
  }

  function expectEscapeCharacter () {
    throw new JSONParseException('JSON_ERROR_0008 Expecting an escape character')
  }

  function expectEscapeUnicode () {
    throw new JSONParseException('JSON_ERROR_0009 Expecting an escape unicode')
  }
}
