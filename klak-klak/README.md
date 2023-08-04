# Klak-Klak

## Language Used

- Zig Version 0.10.1

## Compiled Binary

- main file inside bin folder
- Compiled with Zig 0.10.1 inside Ubuntu 22.04

## Build and run

Build program

`zig build-exe main.zig -O ReleaseFast`

Run program

`./main`

## How to Use

- Run the program
- Enter mathematical expression without parentheses in one line and space separated. Example: `1 + 2 * 3 / -4`.
- Enter empty line to exit program
- Available operations: +, -, *, and /. No parentheses. Negative number allowed.

## Test

Basic operation result

```
10 + 7 = 17
10 - 7 = 3
10 * 7 = 70
10 / 7 = 1
100 + -7 = 93
100 - -7 = 107
100 * -7 = -700
100 / -7 = -14
-100 + 7 = -93
-100 - 7 = -107
-100 * 7 = -700
-100 / 7 = -14
-100 + -7 = -107
-100 - -7 = -93
-100 * -7 = 700
-100 / -7 = 14
```

Another result

```
1 + 2 * -3 / -4 = 2
```