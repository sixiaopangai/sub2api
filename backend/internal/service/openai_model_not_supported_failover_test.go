package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type openAIModelUnsupportedRepoStub struct {
	AccountRepository
	calls []openAIModelUnsupportedRepoCall
}

type openAIModelUnsupportedRepoCall struct {
	accountID int64
	modelKey  string
	resetAt   time.Time
}

func (r *openAIModelUnsupportedRepoStub) SetModelRateLimit(ctx context.Context, id int64, scope string, resetAt time.Time) error {
	r.calls = append(r.calls, openAIModelUnsupportedRepoCall{accountID: id, modelKey: scope, resetAt: resetAt})
	return nil
}

func TestOpenAIModelUnsupportedResponseTriggersFailover(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := &OpenAIGatewayService{}
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-5.5"}`))

	resp := &http.Response{
		StatusCode: http.StatusBadRequest,
		Header:     http.Header{},
		Body:       ioNopCloser(`{"detail":"The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account."}`),
	}
	account := &Account{ID: 213, Name: "free", Platform: PlatformOpenAI, Type: AccountTypeOAuth}

	_, err := svc.handleErrorResponse(c.Request.Context(), resp, c, account, []byte(`{"model":"gpt-5.5"}`))
	require.Error(t, err)
	var failoverErr *UpstreamFailoverError
	require.ErrorAs(t, err, &failoverErr)
	require.True(t, failoverErr.ModelUnsupported)
	require.Equal(t, "gpt-5.5", failoverErr.ModelUnsupportedKey)
	require.False(t, c.Writer.Written(), "failover path must not write the final client error before account switch")
}

func TestOpenAIAccountEligibilitySkipsModelRateLimitedAccount(t *testing.T) {
	resetAt := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
	account := &Account{
		ID:          1,
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Status:      StatusActive,
		Schedulable: true,
		Credentials: map[string]any{"model_mapping": map[string]any{"gpt-5.5": "gpt-5.5"}},
		Extra: map[string]any{
			"model_rate_limits": map[string]any{
				"gpt-5.5": map[string]any{"rate_limit_reset_at": resetAt},
			},
		},
	}

	require.False(t, isOpenAIAccountEligibleForRequest(account, "gpt-5.5", false))
}

func TestOpenAIModelUnsupportedFailoverMarksModelAndClearsSticky(t *testing.T) {
	repo := &openAIModelUnsupportedRepoStub{}
	cache := &schedulerTestGatewayCache{sessionBindings: map[string]int64{}}
	svc := &OpenAIGatewayService{accountRepo: repo, cache: cache}
	groupID := int64(2)
	sessionHash := "session-abc"
	account := &Account{
		ID:          213,
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Status:      StatusActive,
		Schedulable: true,
		Credentials: map[string]any{"model_mapping": map[string]any{"gpt-5.5": "gpt-5.5"}},
	}
	require.NoError(t, svc.setStickySessionAccountID(context.Background(), &groupID, sessionHash, account.ID, time.Hour))

	svc.HandleOpenAIModelUnsupportedFailover(context.Background(), &groupID, sessionHash, account, "gpt-5.5", "gpt-5.5")

	require.Len(t, repo.calls, 1)
	require.Equal(t, int64(213), repo.calls[0].accountID)
	require.Equal(t, "gpt-5.5", repo.calls[0].modelKey)
	require.Greater(t, time.Until(repo.calls[0].resetAt), time.Minute)
	require.NotContains(t, cache.sessionBindings, svc.openAISessionCacheKey(sessionHash))
	require.GreaterOrEqual(t, cache.deletedSessions[svc.openAISessionCacheKey(sessionHash)], 1)
}

func ioNopCloser(body string) *nopReadCloser {
	return &nopReadCloser{Reader: strings.NewReader(body)}
}

type nopReadCloser struct {
	*strings.Reader
}

func (n *nopReadCloser) Close() error { return nil }
