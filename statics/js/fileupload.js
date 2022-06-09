fileChange(e) {
    let file = e.target.files[0] if (file) {
        const fileSize = file.size
        const isLt3G = fileSize / 1024 / 1024 < 5
        if (!isLt3G) {
            this.$message({message : '上传文件大小不得超过5M', type : 'error'})
            file = ''
            return
        }
        const name = file.name
        const fileName = name.substring(name.lastIndexOf('.') + 1).toLowerCase()
        if (fileName !== 'jpg' && fileName !== 'jpeg' && fileName !== 'png') {
            this.$message({
                message : '请选择格式文件(jpg/jpeg/png)上传！',
                type : 'error'
            })
            file = ''
            return
        }
    }
}
,
