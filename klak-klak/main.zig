const std = @import("std");
const print = @import("std").debug.print;

pub fn adder(a: i32, b: i32) i32 {
    var sum_without_carry: i32 = 0;
    var carry: i32 = 0;
    var result: i32 = a;
    var num: i32 = b;

    while (num != 0) {
        sum_without_carry = result ^ num;
        carry = (result & num) << 1;
        result = sum_without_carry;
        num = carry;
    }

    return result;
}

pub fn subtractor(a: i32, b: i32) i32 {
    return adder(a, adder(~b, 1));
}

pub fn multiplier_divider(_a: i32, _b: i32, is_multiply: bool) i32 {
    var a: i32 = _a;
    var b: i32 = _b;

    if ((b < 0) and (a < 0)) {
        a = adder(~a, 1);
        b = adder(~b, 1);
    }

    var remaining_multiplier = b;
    var remaining_div = a;
    var is_negative_div = false;

    if (is_multiply) {
        if ((b == 0) or (a == 0)) {
            return 0;
        } else if (b < 0 and a > 0) {
            var temp = a;
            remaining_multiplier = temp;
            a = b;
            b = temp;
        }
    } else {
        if (a == 0 or b == 0) {
            return 0;
        }

        if (a < 0 and b > 0) {
            is_negative_div = true;
            a = adder(~a, 1);
            remaining_div = a;
        } else if (a > 0 and b < 0) {
            is_negative_div = true;
            b = adder(~b, 1);
        }
    }

    var result: i32 = 0;

    while ((remaining_multiplier != 0 and is_multiply) or (remaining_div > b and !is_multiply)) {
        if (is_multiply) {
            result = adder(result, a);
            remaining_multiplier = subtractor(remaining_multiplier, 1);
        } else {
            result = adder(result, 1);
            remaining_div = subtractor(remaining_div, b);
        }
    }

    if (!is_multiply and is_negative_div) {
        return adder(~result, 1);
    }
    return result;
}

pub fn multiply(a: i32, b: i32) i32 {
    return multiplier_divider(a, b, true);
}

pub fn divide(a: i32, b: i32) i32 {
    return multiplier_divider(a, b, false);
}

pub fn basic_test() !void {
    {
        const a = 10;
        const b = 7;

        print("{} + {} = {}\n", .{ a, b, adder(a, b) });
        print("{} - {} = {}\n", .{ a, b, subtractor(a, b) });
        print("{} * {} = {}\n", .{ a, b, multiply(a, b) });
        print("{} / {} = {}\n", .{ a, b, divide(a, b) });
    }

    {
        const a = 100;
        const b = -7;

        print("{} + {} = {}\n", .{ a, b, adder(a, b) });
        print("{} - {} = {}\n", .{ a, b, subtractor(a, b) });
        print("{} * {} = {}\n", .{ a, b, multiply(a, b) });
        print("{} / {} = {}\n", .{ a, b, divide(a, b) });
    }

    {
        const a = -100;
        const b = 7;

        print("{} + {} = {}\n", .{ a, b, adder(a, b) });
        print("{} - {} = {}\n", .{ a, b, subtractor(a, b) });
        print("{} * {} = {}\n", .{ a, b, multiply(a, b) });
        print("{} / {} = {}\n", .{ a, b, divide(a, b) });
    }

    {
        const a = -100;
        const b = -7;

        print("{} + {} = {}\n", .{ a, b, adder(a, b) });
        print("{} - {} = {}\n", .{ a, b, subtractor(a, b) });
        print("{} * {} = {}\n", .{ a, b, multiply(a, b) });
        print("{} / {} = {}\n", .{ a, b, divide(a, b) });
    }
}

pub fn main() !void {
    const stdin = std.io.getStdIn().reader();
    print("Welcome to klak-klak calculator\n", .{});
    while (true) {
        print("Write expression (space separated), example: 1 + 2 * 3 / 4. Enter empty line to exit\n", .{});
        var buffer: [1024]u8 = undefined;

        if (try stdin.readUntilDelimiterOrEof(buffer[0..], '\n')) |value| {
            if (value.len == 0) {
                print("Exiting\n", .{});
                break;
            }

            var it = std.mem.split(u8, value, " ");

            var result: i32 = 0;
            var has_first_number = false;
            var expect_number = true;
            var ops = "?";

            while (it.next()) |x| {
                if (!has_first_number) {
                    const parsed = try std.fmt.parseInt(i32, x, 10);
                    result = parsed;
                    has_first_number = true;
                    expect_number = false;
                } else if (expect_number) {
                    const parsed = try std.fmt.parseInt(i32, x, 10);

                    if (std.mem.eql(u8, ops, "+")) {
                        result = adder(result, parsed);
                    } else if (std.mem.eql(u8, ops, "-")) {
                        result = subtractor(result, parsed);
                    } else if (std.mem.eql(u8, ops, "*")) {
                        result = multiply(result, parsed);
                    } else if (std.mem.eql(u8, ops, "/")) {
                        result = divide(result, parsed);
                    } else {
                        print("Invalid ops found\n", .{});
                    }

                    expect_number = false;
                } else {
                    expect_number = true;
                    if (std.mem.eql(u8, x, "+")) {
                        ops = "+";
                    } else if (std.mem.eql(u8, x, "-")) {
                        ops = "-";
                    } else if (std.mem.eql(u8, x, "*")) {
                        ops = "*";
                    } else if (std.mem.eql(u8, x, "/")) {
                        ops = "/";
                    } else {
                        print("Invalid ops found\n", .{});
                    }
                }
            }

            print("Result {}\n", .{result});
        } else {
            print("Failed to read line. Exiting\n", .{});
            break;
        }
    }
}
