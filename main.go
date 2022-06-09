package main

import (
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type User struct {
	ID string `gorm:"primarykey"`

	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`

	Name string
	Age  uint8
	Res  string
}

func main() {
	// 连接数据库
	db, err := gorm.Open(sqlite.Open("./db/project_ddd.db"))
	if err != nil {
		log.Fatal("open db failed!")
	}

	// 对应
	db.AutoMigrate(&User{})

	// 创建

	// ---------------------------------------------------------------
	r := gin.Default()

	r.Static("/static", "./statics")
	r.Static("/css", "./statics/css")
	r.Static("/img", "./statics/img")
	r.Static("/js", "./statics/js")
	r.Static("/fonts", "./statics/fonts")

	r.LoadHTMLGlob("./templates/*")

	// ---------------------------------------------------------------
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	r.GET("/index.html", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	r.GET("/about.html", func(c *gin.Context) {
		c.HTML(http.StatusOK, "about.html", nil)
	})

	r.GET("/about", func(c *gin.Context) {
		c.HTML(http.StatusOK, "about.html", nil)
	})

	r.GET("/service.html", func(c *gin.Context) {
		c.HTML(http.StatusOK, "service.html", nil)
	})

	// ---------------- 查询----------------------

	// -----------------取结果---------------------
	r.POST("/service.html", func(c *gin.Context) {
		name := c.PostForm("name")
		if name != "" {
			age := c.PostForm("age")

			file, err := c.FormFile("f1")
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"message": err.Error(),
				})
				return
			}

			// 判断是不是图片，最简单的做法，最好的做法是判断文件前面的字节
			filename := file.Filename
			filesuffix := path.Ext(filename)
			if filesuffix != ".jpg" && filesuffix != ".png" && filesuffix != ".jpeg" {
				c.JSON(http.StatusOK, gin.H{
					"文件类型错误": "请上传后缀为jpeg/jpg/png的文件",
				})
				return
			}

			// 重命名文件
			id := strings.TrimSuffix(getUUID(), "\n")
			newFileName := id + filesuffix
			dst := "./images/" + newFileName

			// 上传文件到指定的目录
			c.SaveUploadedFile(file, dst)
			c.HTML(http.StatusOK, "res2.html", gin.H{
				"ID": id,
			})

			// 将信息写入数据库
			agee, _ := strconv.Atoi(age)

			// ---------------同步：得到python结果--------------------------
			type output struct {
				out []byte
				err error
			}

			ch := make(chan output)

			go func() {
				cmd := exec.Command("python", "dog.py", dst)
				out, err := cmd.CombinedOutput()
				ch <- output{out, err}
			}()

			select {
			case <-time.After(5 * time.Second):
				fmt.Println("timed out")
			case x := <-ch:
				user := User{Name: name, Age: uint8(agee), ID: id, Res: string(x.out)}
				if x.err != nil {
					fmt.Printf("python exec errored: %s\n", x.err)
				}
				if err := db.Create(&user).Error; err != nil {
					log.Fatalf("插入数据失败：%v", err)
				}
			}

			// -----------------------------------------------

		} else {
			uuid := c.PostForm("uuid")

			// 如果查不到，就return
			fmt.Println("uuid:", uuid)
			var user User
			if err := db.First(&user, "id = ?", uuid).Error; err != nil {
				c.JSON(http.StatusOK, gin.H{
					"错误": "请输入正确的凭证码",
				})
				return
			}

			time := user.CreatedAt
			name := user.Name
			age := user.Age
			res := user.Res

			c.HTML(http.StatusOK, "res.html", gin.H{
				"Time": time,
				"Name": name,
				"Age":  age,
				"ID":   uuid,
				"Res":  res,
			})
		}
	})

	r.Run()
}

// 使用linux命令生成uuid
func getUUID() string {
	out, err := exec.Command("uuidgen").Output()
	if err != nil {
		log.Fatal("uuid gen failed")
	}

	return string(out)
}

func getRes(dst string) string {
	out, err := exec.Command("python dog.py " + dst).Output()
	if err != nil {
		log.Fatal("res get failed! err:", err)
	}

	return string(out)
}
