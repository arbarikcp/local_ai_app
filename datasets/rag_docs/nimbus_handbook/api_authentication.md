# API Authentication

The Nimbus API uses OAuth 2.0 bearer tokens. Generate a personal access token in Account
Settings > Developer > API Tokens; tokens can be scoped to read-only or read-write access.
Personal access tokens do not expire unless manually revoked, so treat them like passwords.

For applications acting on behalf of other users, use the standard OAuth 2.0 authorization
code flow; Nimbus does not support the implicit grant flow for new API integrations
registered after 2024. All API requests must be made over HTTPS; plain HTTP requests are
rejected.
