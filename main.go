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

var (
	user       User
	filesuffix string
	mlName     = "dog.py"
)

func main() {
	// ---------------------- 连接数据库 -------------------------------
	db, err := gorm.Open(sqlite.Open("./db/project_ddd.db"))
	if err != nil {
		log.Fatal("open db failed!")
	}

	// 对应
	db.AutoMigrate(&User{})

	// ------------------------静态文件----------------------------------
	r := gin.Default()

	r.Static("/static", "./statics")
	r.Static("/css", "./statics/css")
	r.Static("/img", "./statics/img")
	r.Static("/js", "./statics/js")
	r.Static("/fonts", "./statics/fonts")
	r.Static("/images/", "./images/")

	r.LoadHTMLGlob("./templates/*")

	// -----------------------路由---------------------------------
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	r.GET("/index.html", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	r.GET("/service.html", func(c *gin.Context) {
		c.HTML(http.StatusOK, "service.html", nil)
	})

	r.GET("/about.html", func(c *gin.Context) {
		c.HTML(http.StatusOK, "about.html", nil)
	})

	r.NoRoute(func(c *gin.Context) {
		c.HTML(http.StatusNotFound, "404.html", nil)
	})

	// ---------------- 查询----------------------

	r.POST("/service.html", func(c *gin.Context) {
		userName := c.PostForm("name")
		if userName != "" {
			// === 一般查询 ===
			age := c.PostForm("age")
			agee, _ := strconv.Atoi(age)

			file, err := c.FormFile("f1")
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"message": err.Error(),
				})
				return
			}
			// 判断文件：是不是图片格式
			filename := file.Filename
			filesuffix = path.Ext(filename)
			if filesuffix != ".jpg" && filesuffix != ".png" && filesuffix != ".jpeg" {
				c.HTML(http.StatusOK, "error.html", gin.H{
					"Error":     "文件类型错误",
					"whatError": "请上传后缀为jpeg/jpg/png的文件",
				})
				return
			}

			// 重命名文件
			id := strings.TrimSuffix(getUUID(), "\n")
			newFileName := id + filesuffix

			// 上传文件 -> 指定的目录
			dst := "./images/" + newFileName
			c.SaveUploadedFile(file, dst)

			// 计算 & 插入数据库
			insertUser(agee, dst, id, userName, db, c)
		} else {
			// === 历史结果查询 ===
			uuid := c.PostForm("uuid")

			// 根据uuid查询
			if err := db.First(&user, "id = ?", uuid).Error; err != nil {
				c.HTML(http.StatusOK, "error.html", gin.H{
					"Error":     "凭证码错误",
					"whatError": "请输入正确的凭证码",
				})
				return
			}

		}

		name := user.Name
		time := user.CreatedAt
		uuid := user.ID
		res := user.Res
		img := "images/" + user.ID + filesuffix

		c.HTML(http.StatusOK, "res.html", gin.H{
			"Name":  name,
			"Time":  time.Format("2006/01/02 15:04"),
			"ID":    uuid,
			"Res":   res,
			"image": img,
		})
	})

	r.Run(":8080")
}

// 使用linux命令生成uuid
func getUUID() string {
	out, err := exec.Command("uuidgen").Output()
	if err != nil {
		log.Fatal("uuid gen failed")
	}

	return string(out)
}

func insertUser(agee int, dst string, id string, userName string, db *gorm.DB, c *gin.Context) {
	type output struct {
		out []byte
		err error
	}

	ch := make(chan output)

	go func() {
		cmd := exec.Command("python", mlName, dst)
		out, err := cmd.CombinedOutput()
		ch <- output{out, err}
	}()

	select {
	case <-time.After(10 * time.Second):
		fmt.Println("timed out")
	case x := <-ch:
		user = User{Name: userName, Age: uint8(agee), ID: id, Res: string(x.out)}
		if x.err != nil {
			c.HTML(http.StatusOK, "error.html", gin.H{
				"Error":     "图片内容错误",
				"whatError": "请上传皮肤图片",
			})
			return
		}
		if err := db.Create(&user).Error; err != nil {
			log.Fatalf("插入数据失败：%v", err)
		}
	}

}
