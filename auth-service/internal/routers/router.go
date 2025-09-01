package routers

import (
	"auth-service/internal/config"
	"auth-service/internal/handlers"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, authHandler *handlers.AuthHandler) *gin.Engine {
	r := gin.Default()

	// Auth routes
	r.POST("/register", authHandler.Register)
	r.POST("/login", authHandler.Login)
	r.POST("/refresh", authHandler.Refresh)
	r.POST("/logout", authHandler.Logout)

	// Protected route sample
	me := r.Group("/me")
	{
		me.GET("", authHandler.Me)
	}

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
