CC = gcc
CFLAGS = -O0 -g -Wall -fsignaling-nans

OBJ_ABEL =  abel_prgrad_test.o prgrad_reg.o prgrad_reg_common.o

OBJ_FREDH =  fredh_prgrad_test.o prgrad_reg.o prgrad_reg_common.o

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
                
fredh_prgrad_test:	$(OBJ_FREDH)
	$(CC) $(CFLAGS) -o fredh_prgrad_test $(OBJ_FREDH) -lm

abel_prgrad_test:	$(OBJ_ABEL)
	$(CC) $(CFLAGS) -o abel_prgrad_test $(OBJ_ABEL) -lm

clean:
	rm -f *~ *.o abel_prgrad_test fredh_prgrad_test
