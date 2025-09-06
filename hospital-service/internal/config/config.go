package config

import (
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	Port             string
	DBUrl            string
	AccessSecret     string
	RefreshSecret    string
	AccessTTLMinutes int
	RefreshTTLDays   int

	RefreshCookieName string
	CookieDomain      string
	CookieSecure      bool
	CookieSameSite    string // lax/strict/none
	S3Bucket string
    S3Region string
    AWSAccessKey string
    AWSSecretKey string
}

func Load() Config {
	_ = godotenv.Load()

	return Config{
		Port:              getEnv("PORT", "8080"),
		DBUrl:             mustEnv("DATABASE_URL"),
		AccessSecret:      mustEnv("JWT_ACCESS_SECRET"),
		RefreshSecret:     mustEnv("JWT_REFRESH_SECRET"),
		AccessTTLMinutes:  getEnvInt("ACCESS_TOKEN_TTL_MIN", 15),
		RefreshTTLDays:    getEnvInt("REFRESH_TOKEN_TTL_DAY", 7),
		RefreshCookieName: getEnv("REFRESH_COOKIE_NAME", "refresh_token"),
		CookieDomain:      getEnv("COOKIE_DOMAIN", "localhost"),
		CookieSecure:      getEnvBool("COOKIE_SECURE", false),
		CookieSameSite:    strings.ToLower(getEnv("COOKIE_SAMESITE", "strict")),
		S3Bucket:   mustEnv("S3_BUCKET"),
        S3Region:   mustEnv("S3_REGION"),
        AWSAccessKey: mustEnv("AWS_ACCESS_KEY_ID"),
        AWSSecretKey: mustEnv("AWS_SECRET_ACCESS_KEY"),
	}
}

func getEnv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
func mustEnv(k string) string {
	v := os.Getenv(k)
	if v == "" {
		log.Fatalf("missing required env: %s", k)
	}
	return v
}
func getEnvInt(k string, d int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return d
}
func getEnvBool(k string, d bool) bool {
	if v := os.Getenv(k); v != "" {
		v = strings.ToLower(v)
		return v == "1" || v == "true" || v == "yes"
	}
	return d
}
