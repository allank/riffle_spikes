security/oauth2
token-refresh.md

# OAuth 2.0 Token Refresh

Describes the refresh token rotation pattern used in single page applications. When an access token expires, the client presents its refresh token to the authorization server's token endpoint and receives a new access token (and, under rotation, a new refresh token) without requiring the user to re-authenticate.

Related: [[OAuth Scopes and Consent]], [[Token Storage]]
