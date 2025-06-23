import s3fs

fs = s3fs.S3FileSystem()
print(fs.ls("sophieperrinlyon2"))
