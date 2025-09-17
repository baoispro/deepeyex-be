package routers

import (
	"auth-service/internal/config"
	"auth-service/internal/handlers"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func SetupRouter(cfg *config.Config, authHandler *handlers.AuthHandler, userHandler *handlers.UserHandler) *gin.Engine {
	r := gin.Default()
	r.Use(cors.Default())

	// ===== Public routes =====
	public := r.Group("/public")
	{
		public.POST("/register", authHandler.Register)
		public.POST("/login", authHandler.Login)
		public.POST("/login/firebase", authHandler.LoginFirebase)
		public.POST("/refresh", authHandler.Refresh)
		public.POST("/logout", authHandler.Logout)
		public.POST("/reset-password", userHandler.UpdatePasswordByEmail)
	}

	// ===== Protected routes (cần JWT) =====
	private := r.Group("/private")
	{
		private.GET("/me", authHandler.Me)

		private.POST("/users", userHandler.Create)
		private.GET("/users", userHandler.List)
		private.GET("/users/:id", userHandler.Get)
		private.PUT("/users/:id", userHandler.Update)
		private.DELETE("/users/:id", userHandler.Delete)
	}

	// Swagger
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	return r
}
