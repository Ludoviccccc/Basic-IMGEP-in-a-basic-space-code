# 8085simCli
8085 sim CLI version

***A simple 8085 CLI compiler that reads hexcodes from a textfile and produces the corresponding output.***



Clone the repository 


``` 
git clone https://github.com/wd400/8085simCli
cd 8085simCli
```

Ensure you have python installed and is set in your PATH

Execute the execute.py with a text file that you want to execute.

```
python execute.py somefile.txt -options

-reg            display registers in the output
-flag           display flag values in the output
-port           display the ports in the output

-start [MEM]    specify a start memory location for the compiler to start execution from.

-read [MEM]     display a certain memory location after the compiler executes.

-output [FILE]   pipe the output to a text file instead of logging onto console.

```

You can get the details of these options using

```python
python execute.py -help
```


Some rules for writing a script file

1. You can only enter hex values of the corresponding opcodes of 8085 instruction set.
2. Every hexcode should be on a new line
3. If a hexcode terminates in a semicolon(:), it is considered to be a memory location. The memory pointer will be reset to this location while writing the hexcodes.
4. If a line starts with '/', it is considered a comment
5. Anything other than the above mentioned things will be thrown as error.


The examples directory list few examples of how a script file should be written.
