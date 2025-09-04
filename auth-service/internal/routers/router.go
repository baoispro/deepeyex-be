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

	// ===== Public routes =====
	public := r.Group("/public")
	{
		public.POST("/register", authHandler.Register)
		public.POST("/login", authHandler.Login)
		public.POST("/login/firebase", authHandler.LoginFirebase)
		public.POST("/refresh", authHandler.Refresh)
		public.POST("/logout", authHandler.Logout)
	}

	// ===== Protected routes (cần JWT) =====
	private := r.Group("/private")
	{
		private.GET("/me", authHandler.Me)
	}

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
