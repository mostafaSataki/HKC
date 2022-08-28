
from hashlib import sha256
from Crypto import Random
from Crypto.Cipher import AES
import sys


BS = 16
pad = lambda s: s + bytes((BS - len(s) % BS) * chr(BS - len(s) % BS), 'utf-8')
unpad = lambda s : s[0:-(s[-1])]


class AESCipher:
    def __init__(self, key):
        self.key = sha256(key.encode('utf-8')).digest()


    def encrypt(self, raw):
        # raw = pad(raw)

        raw = pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)


    def decrypt(self, enc):
        iv = enc[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(enc[16:]))

    def _read_bin_file(self,filename):
        try:
            fp = open(src_file, "rb")
            data = fp.read()
            fp.close()
        except:
            print("Error IO on \"" + src_file + "\"")
            exit()
        return data

    def _write_bin_file(self, filename, data):
        try:
            fp = open(filename, "wb")
            fp.write(data)
            fp.close()
        except:
            print("Error IO on \"" + filename + "\"")
            exit()

    def encrypt_data(self, data):

        if option == "enc":
            if data == "":
                data = "0"
            else:
                data = self.encrypt(data)
        return data

    def decrypt_data(self, data):

        if data == "0":
            data = ""
        else:
            data = self.decrypt(data)
            if data == "":
                print("Bad key.")
                exit()
        return data

    def encrypt_file(self,src_file,dst_file ):
        data = self._read_bin_file(src_file)
        data = self.encrypt_data(data)
        self._write_bin_file(dst_file, data)

    def decrypt_file(self,src_file,dst_file ):
        data = self._read_bin_file(src_file)

        data = self.decrypt_data(data)
        self._write_bin_file(dst_file, data)


