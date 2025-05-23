# Contributing to cuHPX

Thank you for your interest in contributing to cuHPX!

Your contributions are greatly appreciated and help improve the project for the community. This guide outlines the process for proposing changes and submitting pull requests.

---

## Workflow

1. **Fork** the cuHPX repository to your GitHub or GitLab account.
2. **Clone** your fork and create a feature branch:
   ```bash
   git checkout -b my-feature
   ```
3. **Install pre-commit hooks** to ensure formatting and linting:
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```
4. **Make your changes**, including:
   - Code or documentation updates
   - Tests or examples if applicable
5. **Sign off your commits** using the `-s` flag:
   ```bash
   git commit -s -m "Add cool feature"
   ```

6. Push your changes and open a **Pull Request** against the main branch of cuHPX.

---

## Developer Certificate of Origin (DCO)

All contributors must "sign off" on their commits to certify compliance with the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

By signing off, you confirm the following:
```
Developer Certificate of Origin
Version 1.1
  
Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129
  
Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
Developer Certificate of Origin
Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

You must include a sign-off line in your commit message, which looks like this:
```
Signed-off-by: Your Name <your.email@nvidia.com>
```
Use the `-s` flag with `git commit` to automatically append it.

---

## License

By contributing to cuHPX, you agree that your contributions will be licensed under the same terms as specified in the [LICENSE.txt](./LICENSE.txt) file.
