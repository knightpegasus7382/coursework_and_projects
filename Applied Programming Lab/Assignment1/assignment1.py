import sys                   # [Module containing the function to accept command-line] arguments
flagc=0; flage=0; ckt=[]; c=0; e=0; count=-1;
if (len(sys.argv)>2):
    print("\nError: Too many command-line arguments/filenames, please enter one filename")
else:    
    try:
        f=open(sys.argv[1])
    except FileNotFoundError:
        print("\nError: Invalid file name")   # [This error is displayed when a file does not exist with the given filename]
    except IndexError:
        print("\nError: File name not entered")   # [This error is displayed when the filename argument is not entered in the command line at all]
    else:
        for line in f: 
            count+=1
            if (line.strip("\n")==".circuit"):
                flagc+=1; c=count
            if (line.strip("\n")==".end"):
                flage+=1; e=count
        f.seek(0)
        if((flagc!=1) | (flage!=1)):
            print("\nError: There should be exactly one '.circuit' and one '.end' for the file to contain a valid circuit")
        else:
            ckt=[line for count,line in enumerate(f) if (count>c and count<e)]
        f.close()
    for j in range(len(ckt)):
        if ("#" in ckt[j]):
            ckt[j]=ckt[j][:ckt[j].index("#")].strip()+"\n"    # [Removes the comments in the circuit]
        ckt[j]=list(reversed((ckt[j].replace("\n"," \n")).split(' ')))      # [Order of the tokens in each line is reversed]
    ckt.reverse()                # [Order of the lines is reversed]
    for w in ckt:
        for x in w:            
            print("%s" % x, end = ' ')
print()
        


  
