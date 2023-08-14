import { JSONStringifyException } from './exception'

export function jsonToString (obj: any): string {
  const isArray = (value: any) => {
    return Array.isArray(value) && typeof value === 'object'
  }

  const isObject = (value: any) => {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
  }

  const isString = (value: any) => {
    return typeof value === 'string'
  }

  const isBoolean = (value: any) => {
    return typeof value === 'boolean'
  }

  const isNumber = (value: any) => {
    return typeof value === 'number'
  }

  // Common check for number, string and boolean value
  const restOfDataTypes = (value: any) => {
    return isNumber(value) || isString(value) || isBoolean(value)
  }

  // Boolean and Number behave in a same way and String we need to add extra qoutes
  if (restOfDataTypes(obj)) {
    const passQuotes = isString(obj) ? '"' : ''
    // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
    return `${passQuotes}${obj}${passQuotes}`
  }

  // This function will be used to remove extra comma from the arrays and object
  const removeComma = (str: string) => {
    const tempArr = str.split('')
    tempArr.pop()
    return tempArr.join('')
  }

  // Recursive function call for Arrays to handle nested arrays
  if (isArray(obj)) {
    let arrStr = '';
    (obj as any[]).forEach((eachValue) => {
      arrStr += jsonToString(eachValue)
      arrStr += ','
    })

    return '[' + removeComma(arrStr) + ']'
  }

  // Recursive function call for Object to handle nested Object
  if (isObject(obj)) {
    let objStr = ''

    const objKeys = Object.keys(obj)

    objKeys.forEach((eachKey) => {
      const eachValue = obj[eachKey]
      objStr += `"${eachKey}":${jsonToString(eachValue)},`
    })
    return '{' + removeComma(objStr) + '}'
  }

  throw new JSONStringifyException('Invalid data type')
};
